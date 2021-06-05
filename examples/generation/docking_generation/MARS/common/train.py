import os
import torch
import logging as log

from ..common.utils import Recorder


def evaluate(model, loader, metrics=['loss']):
    model.eval()
    recoder = Recorder(metrics)
    with torch.no_grad():
        for batch in loader:
            batch_size, metric_values = \
                model.loss(batch, metrics)
            recoder.record(batch_size, metric_values)
    return recoder.report_avg()

def train(model, loaders, optimizer,
        n_epoch=200,
        max_step=0,
        log_every=0,
        eval_every=0,
        save_dir=None,
        writer=None,
        metrics=['loss']
    ):
    log.info("training...")
    recorder = Recorder(metrics)
    best_eval_loss = 10.
    
    step = 0
    for epoch in range(n_epoch):
        log.info('Epoch: {:03d}'.format(epoch))

        for batch in loaders['dev']:
            if max_step > 0 and step >= max_step:
                break
            
            model.train()
            optimizer.zero_grad()
            batch_size, metric_values = \
                model.loss(batch, metrics)

            loss = metric_values[0]
            loss.backward()
            optimizer.step()
            
            step += 1
            recorder.record(batch_size, metric_values)

            def log_records(split, metric2avg):
                log_str = 'step: %03dk (%s)' % (step // 1000, split)
                for metric, avg in metric2avg.items():
                    log_str += ', %s: %.3f' % (metric, avg)
                    if writer: writer.add_scalar(
                        '%s/%s' % (split, metric), avg, step)
                log.info(log_str)
            
            if log_every > 0 and step % log_every == 0:
                metric2avg = recorder.report_avg()
                log_records('dev', metric2avg)
                recorder.reset()

            if eval_every > 0 and step % eval_every == 0:
                metric2avg = evaluate(model, loaders['val'], metrics)
                log_records('val', metric2avg)
                loss = metric2avg['loss']
                if save_dir and loss < best_eval_loss:
                    best_eval_loss = loss
                    torch.save(model.state_dict(), 
                        os.path.join(save_dir, 'model_best.pt'))
